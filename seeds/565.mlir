module {
  func.func @main(%arg0: tensor<5xf32>, %arg1: tensor<94x48x35x24xf32>, %arg2: tensor<96x52x61x4xf32>, %arg3: tensor<96xf32>) -> (tensor<8xf32>, tensor<94x102x98x96xi1>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<5xf32>) -> tensor<1xf32>
    %1 = tosa.add %0, %0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.sigmoid %1 : (tensor<1xf32>) -> tensor<1xf32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 94, 102, 98, 96>} : (tensor<94x48x35x24xf32>, tensor<96x52x61x4xf32>, tensor<96xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<94x102x98x96xf32>
    %4 = tosa.tanh %2 : (tensor<1xf32>) -> tensor<1xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_5_size = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<1xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<8xf32>
    %6 = tosa.greater_equal %3, %3 : (tensor<94x102x98x96xf32>, tensor<94x102x98x96xf32>) -> tensor<94x102x98x96xi1>
    %7 = tosa.exp %5 : (tensor<8xf32>) -> tensor<8xf32>
    %8 = tosa.logical_not %6 : (tensor<94x102x98x96xi1>) -> tensor<94x102x98x96xi1>
    return %7, %8 : tensor<8xf32>, tensor<94x102x98x96xi1>
  }
}
