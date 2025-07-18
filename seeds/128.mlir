module {
  func.func @main(%arg0: tensor<39x35x56xf32>, %arg1: tensor<21x47x78x88xf32>, %arg2: tensor<58x100x80x42xf32>, %arg3: tensor<58xf32>) -> (tensor<39x35x56xi1>, tensor<39x35x56xf32>, tensor<21x149x237x58xf32>) {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<39x35x56xf32>) -> tensor<39x35x56xf32>
    %1 = tosa.equal %0, %0 : (tensor<39x35x56xf32>, tensor<39x35x56xf32>) -> tensor<39x35x56xi1>
    %2 = tosa.log %0 : (tensor<39x35x56xf32>) -> tensor<39x35x56xf32>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 21, 149, 237, 58>} : (tensor<21x47x78x88xf32>, tensor<58x100x80x42xf32>, tensor<58xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<21x149x237x58xf32>
    return %1, %2, %3 : tensor<39x35x56xi1>, tensor<39x35x56xf32>, tensor<21x149x237x58xf32>
  }
}
