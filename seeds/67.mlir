module {
  func.func @main(%arg0: tensor<26x52x4xi1>, %arg1: tensor<62x68x62x23xf32>, %arg2: tensor<58x98x10x4xf32>, %arg3: tensor<58xf32>) -> (tensor<26x52x1xi1>, tensor<62x236x74x58xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 2 : i32} : (tensor<26x52x4xi1>) -> tensor<26x52x1xi1>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 2, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 62, 236, 74, 58>} : (tensor<62x68x62x23xf32>, tensor<58x98x10x4xf32>, tensor<58xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<62x236x74x58xf32>
    %2 = tosa.bitwise_not %0 : (tensor<26x52x1xi1>) -> tensor<26x52x1xi1>
    %3 = tosa.sigmoid %1 : (tensor<62x236x74x58xf32>) -> tensor<62x236x74x58xf32>
    return %2, %3 : tensor<26x52x1xi1>, tensor<62x236x74x58xf32>
  }
}
