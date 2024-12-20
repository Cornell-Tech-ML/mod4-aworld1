from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Custom njit function that always inlines the given function."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    # s1 = input_strides
    # s2 = weight_strides

    s_in_b, s_in_ic, s_in_w = input_strides
    s_wt_oc, s_wt_ic, s_wt_k = weight_strides
    s_out_b, s_out_oc, s_out_w = out_strides

    # Parallelize over out tensor size if possible
    for b in prange(batch_):
        for oc in range(out_channels):
            for w_out in range(out_width):
                acc = 0.0
                for ic in range(in_channels):
                    for k in range(kw):
                        if reverse:
                            # Kernel anchored to the right: input index w_in = w_out - k
                            w_in = w_out - k
                        else:
                            # Kernel anchored to the left: input index w_in = w_out + k
                            w_in = w_out + k

                        # Check boundary conditions (zero-padding)
                        if 0 <= w_in < width:
                            in_pos = b * s_in_b + ic * s_in_ic + w_in * s_in_w
                            wt_pos = oc * s_wt_oc + ic * s_wt_ic + k * s_wt_k
                            acc += input[in_pos] * weight[wt_pos]

                out_pos = b * s_out_b + oc * s_out_oc + w_out * s_out_w
                out[out_pos] = acc


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradients of the Conv1dFun forward pass.

        Args:
        ----
            ctx (Context): The context with saved tensors.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and weight.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # s1 = input_strides
    # s2 = weight_strides
    # inners
    # s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    # s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # Unpack strides for clarity
    s_in_b, s_in_ic, s_in_h, s_in_w = input_strides
    s_wt_oc, s_wt_ic, s_wt_kh, s_wt_kw = weight_strides
    s_out_b, s_out_oc, s_out_h, s_out_w = out_strides

    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, k_height, k_width = weight_shape

    # Parallelize over the output size
    for b in prange(batch_):
        for oc in range(out_channels):
            for h_out in range(out_height):
                for w_out in range(out_width):
                    acc = 0.0
                    for ic in range(in_channels):
                        for kh in range(k_height):
                            for kw_i in range(k_width):
                                if reverse:
                                    # Kernel anchored at bottom-right
                                    h_in = h_out - kh
                                    w_in = w_out - kw_i
                                else:
                                    # Kernel anchored at top-left
                                    h_in = h_out + kh
                                    w_in = w_out + kw_i

                                # Check boundaries
                                if 0 <= h_in < height and 0 <= w_in < width:
                                    in_pos = (
                                        b * s_in_b
                                        + ic * s_in_ic
                                        + h_in * s_in_h
                                        + w_in * s_in_w
                                    )
                                    wt_pos = (
                                        oc * s_wt_oc
                                        + ic * s_wt_ic
                                        + kh * s_wt_kh
                                        + kw_i * s_wt_kw
                                    )
                                    acc += input[in_pos] * weight[wt_pos]

                    out_pos = (
                        b * s_out_b + oc * s_out_oc + h_out * s_out_h + w_out * s_out_w
                    )
                    out[out_pos] = acc


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradients of the Conv2dFun forward pass.

        Args:
        ----
            ctx (Context): The context with saved tensors.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and weight.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
