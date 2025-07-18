module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<91x8xf32>, %arg2: tensor<16x50x48x47xf32>, %arg3: tensor<26x53x17x91xf32>, %arg4: tensor<26xf32>) -> (tensor<91x1xf32>, tensor<16x1x66x26xf32>, tensor<16x1x66x26xf32>, tensor<528x8008xf32>, tensor<i1>) {
    %0 = tosa.floor %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_min %arg1 {axis = 1 : i32} : (tensor<91x8xf32>) -> tensor<91x1xf32>
    %2 = tosa.equal %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 16, 154, 66, 26>} : (tensor<16x50x48x47xf32>, tensor<26x53x17x91xf32>, tensor<26xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<16x154x66x26xf32>
    %4 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<16x154x66x26xf32>) -> tensor<16x1x66x26xf32>
    %5 = tosa.maximum %4, %4 : (tensor<16x1x66x26xf32>, tensor<16x1x66x26xf32>) -> tensor<16x1x66x26xf32>
    %6 = tosa.minimum %5, %5 : (tensor<16x1x66x26xf32>, tensor<16x1x66x26xf32>) -> tensor<16x1x66x26xf32>
    %7 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<16x154x66x26xf32>) -> tensor<16x1x66x26xf32>
    %r_8 = tosa.const_shape {values = dense<[ 528, 8008 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %8 = tosa.reshape %3, %r_8 : (tensor<16x154x66x26xf32>, !tosa.shape<2>) -> tensor<528x8008xf32>
    %9 = tosa.reverse %8 {axis = 1 : i32} : (tensor<528x8008xf32>) -> tensor<528x8008xf32>
    %10 = tosa.ceil %9 : (tensor<528x8008xf32>) -> tensor<528x8008xf32>
    %11 = tosa.clz %2 : (tensor<i1>) -> tensor<i1>
    return %1, %6, %7, %10, %11 : tensor<91x1xf32>, tensor<16x1x66x26xf32>, tensor<16x1x66x26xf32>, tensor<528x8008xf32>, tensor<i1>
  }
}
