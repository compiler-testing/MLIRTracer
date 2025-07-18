module {
  func.func @main(%arg0: tensor<41x36x56x16xf32>, %arg1: tensor<49x47x75x66xf32>, %arg2: tensor<49xf32>, %arg3: tensor<28x31x84x21x23x87xi8>, %arg4: tensor<28x1x84x21x1x87xi8>) -> (tensor<28x31x84x21x23x87xi8>, tensor<41x120x188x49xi1>, tensor<41x120x188x49xf32>, tensor<41x120x188x49xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 41, 120, 188, 49>} : (tensor<41x36x56x16xf32>, tensor<49x47x75x66xf32>, tensor<49xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<41x120x188x49xf32>
    %1 = tosa.rsqrt %0 : (tensor<41x120x188x49xf32>) -> tensor<41x120x188x49xf32>
    %2 = tosa.sub %1, %0 : (tensor<41x120x188x49xf32>, tensor<41x120x188x49xf32>) -> tensor<41x120x188x49xf32>
    %3 = tosa.logical_left_shift %arg3, %arg4 : (tensor<28x31x84x21x23x87xi8>, tensor<28x1x84x21x1x87xi8>) -> tensor<28x31x84x21x23x87xi8>
    %4 = tosa.equal %2, %2 : (tensor<41x120x188x49xf32>, tensor<41x120x188x49xf32>) -> tensor<41x120x188x49xi1>
    %5 = tosa.reciprocal %0 : (tensor<41x120x188x49xf32>) -> tensor<41x120x188x49xf32>
    %6 = tosa.reciprocal %2 : (tensor<41x120x188x49xf32>) -> tensor<41x120x188x49xf32>
    return %3, %4, %5, %6 : tensor<28x31x84x21x23x87xi8>, tensor<41x120x188x49xi1>, tensor<41x120x188x49xf32>, tensor<41x120x188x49xf32>
  }
}
