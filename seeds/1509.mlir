module {
  func.func @main(%arg0: tensor<63x84x60xi32>, %arg1: tensor<35xi1>, %arg2: tensor<1xi1>, %arg3: tensor<21x14x59x17xf32>, %arg4: tensor<67x47x14x59xf32>, %arg5: tensor<67xf32>) -> (tensor<189x84x180xi32>, tensor<35xi1>, tensor<21x62x134x67xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<63x84x60xi32>, !tosa.shape<3>) -> tensor<189x84x180xi32>
    %1 = tosa.intdiv %0, %0 : (tensor<189x84x180xi32>, tensor<189x84x180xi32>) -> tensor<189x84x180xi32>
    %2 = tosa.logical_xor %arg1, %arg2 : (tensor<35xi1>, tensor<1xi1>) -> tensor<35xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 1, 2>, out_shape = array<i64: 21, 62, 134, 67>} : (tensor<21x14x59x17xf32>, tensor<67x47x14x59xf32>, tensor<67xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<21x62x134x67xf32>
    return %1, %2, %3 : tensor<189x84x180xi32>, tensor<35xi1>, tensor<21x62x134x67xf32>
  }
}
