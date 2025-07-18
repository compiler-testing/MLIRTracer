module {
  func.func @main(%arg0: tensor<29xi64>, %arg1: tensor<87x31x43x13xf32>, %arg2: tensor<10x26x61x52xf32>, %arg3: tensor<10xf32>, %arg4: tensor<i1>, %arg5: tensor<i1>) -> (tensor<1x1x1xi64>, tensor<1xi64>, tensor<i1>, tensor<87x59x106x10xi1>, tensor<9x2x9x9xi1>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<29xi64>) -> tensor<1xi64>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<1xi64>, !tosa.shape<3>) -> tensor<1x1x1xi64>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 87, 59, 106, 10>} : (tensor<87x31x43x13xf32>, tensor<10x26x61x52xf32>, tensor<10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<87x59x106x10xf32>
    %3 = tosa.log %2 : (tensor<87x59x106x10xf32>) -> tensor<87x59x106x10xf32>
    %4 = tosa.bitwise_and %0, %0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %5 = tosa.rsqrt %3 : (tensor<87x59x106x10xf32>) -> tensor<87x59x106x10xf32>
    %6 = tosa.logical_and %arg4, %arg5 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.equal %2, %2 : (tensor<87x59x106x10xf32>, tensor<87x59x106x10xf32>) -> tensor<87x59x106x10xi1>
    %8 = tosa.greater %5, %2 : (tensor<87x59x106x10xf32>, tensor<87x59x106x10xf32>) -> tensor<87x59x106x10xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 54, 34, 24, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_9_size = tosa.const_shape {values = dense<[ 9, 2, 9, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %9 = tosa.slice %7, %s_9_start, %s_9_size : (tensor<87x59x106x10xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<9x2x9x9xi1>
    return %1, %4, %6, %8, %9 : tensor<1x1x1xi64>, tensor<1xi64>, tensor<i1>, tensor<87x59x106x10xi1>, tensor<9x2x9x9xi1>
  }
}
