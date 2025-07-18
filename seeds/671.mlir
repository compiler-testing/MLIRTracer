module {
  func.func @main(%arg0: tensor<67xi1>, %arg1: tensor<24x21x43xi8>, %arg2: tensor<1x1x1xi8>, %arg3: tensor<f32>) -> (tensor<2x1x1xi1>, tensor<f32>, tensor<72x21x86xi1>, tensor<f32>, tensor<24x21x43xi1>, tensor<24x21x43xi1>, tensor<f32>) {
    %0 = tosa.clz %arg0 : (tensor<67xi1>) -> tensor<67xi1>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<67xi1>) -> tensor<1xi1>
    %2 = tosa.greater %arg1, %arg2 : (tensor<24x21x43xi8>, tensor<1x1x1xi8>) -> tensor<24x21x43xi1>
    %3 = tosa.logical_xor %2, %2 : (tensor<24x21x43xi1>, tensor<24x21x43xi1>) -> tensor<24x21x43xi1>
    %4 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
    %5 = tosa.tanh %arg3 : (tensor<f32>) -> tensor<f32>
    %r_6 = tosa.const_shape {values = dense<[ 2, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.reshape %4, %r_6 : (tensor<2xi1>, !tosa.shape<3>) -> tensor<2x1x1xi1>
    %7 = tosa.logical_not %6 : (tensor<2x1x1xi1>) -> tensor<2x1x1xi1>
    %8 = tosa.sigmoid %5 : (tensor<f32>) -> tensor<f32>
    %t_9 = tosa.const_shape {values = dense<[ 3, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.tile %3, %t_9 : (tensor<24x21x43xi1>, !tosa.shape<3>) -> tensor<72x21x86xi1>
    %10 = tosa.logical_or %9, %9 : (tensor<72x21x86xi1>, tensor<72x21x86xi1>) -> tensor<72x21x86xi1>
    %11 = tosa.arithmetic_right_shift %10, %10 {round = true} : (tensor<72x21x86xi1>, tensor<72x21x86xi1>) -> tensor<72x21x86xi1>
    %12 = tosa.tanh %5 : (tensor<f32>) -> tensor<f32>
    %13 = tosa.logical_right_shift %3, %3 : (tensor<24x21x43xi1>, tensor<24x21x43xi1>) -> tensor<24x21x43xi1>
    %14 = tosa.add %3, %2 : (tensor<24x21x43xi1>, tensor<24x21x43xi1>) -> tensor<24x21x43xi1>
    %15 = tosa.floor %5 : (tensor<f32>) -> tensor<f32>
    %16 = tosa.sigmoid %5 : (tensor<f32>) -> tensor<f32>
    %17 = tosa.pow %16, %15 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %7, %8, %11, %12, %13, %14, %17 : tensor<2x1x1xi1>, tensor<f32>, tensor<72x21x86xi1>, tensor<f32>, tensor<24x21x43xi1>, tensor<24x21x43xi1>, tensor<f32>
  }
}
