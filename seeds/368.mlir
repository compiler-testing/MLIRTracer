module {
  func.func @main(%arg0: tensor<21x64x43xi32>, %arg1: tensor<21x1x1xi32>, %arg2: tensor<f32>) -> (tensor<42x192x43xi32>, tensor<f32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<21x64x43xi32>, tensor<21x1x1xi32>) -> tensor<21x64x43xi32>
    %1 = tosa.sigmoid %arg2 : (tensor<f32>) -> tensor<f32>
    %t_2 = tosa.const_shape {values = dense<[ 2, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %0, %t_2 : (tensor<21x64x43xi32>, !tosa.shape<3>) -> tensor<42x192x43xi32>
    %3 = tosa.ceil %1 : (tensor<f32>) -> tensor<f32>
    return %2, %3 : tensor<42x192x43xi32>, tensor<f32>
  }
}
