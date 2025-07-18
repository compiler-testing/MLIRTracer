module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<25xf32>, %arg2: tensor<63xf32>, %arg3: tensor<42x74x41x86xi16>, %arg4: tensor<42x1x1x86xi16>) -> (tensor<f32>, tensor<88xf32>, tensor<42x74x41x86xi16>, tensor<88xf32>, tensor<74x41x86xi32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.concat %arg1, %arg2 {axis = 0 : i32} : (tensor<25xf32>, tensor<63xf32>) -> tensor<88xf32>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<88xf32>) -> tensor<88xf32>
    %3 = tosa.logical_right_shift %arg3, %arg4 : (tensor<42x74x41x86xi16>, tensor<42x1x1x86xi16>) -> tensor<42x74x41x86xi16>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<42x74x41x86xi16>, tensor<42x74x41x86xi16>) -> tensor<42x74x41x86xi16>
    %5 = tosa.clz %3 : (tensor<42x74x41x86xi16>) -> tensor<42x74x41x86xi16>
    %6 = tosa.reciprocal %1 : (tensor<88xf32>) -> tensor<88xf32>
    %7 = tosa.argmax %4 {axis = 0 : i32} : (tensor<42x74x41x86xi16>) -> tensor<74x41x86xi32>
    %8 = tosa.bitwise_and %7, %7 : (tensor<74x41x86xi32>, tensor<74x41x86xi32>) -> tensor<74x41x86xi32>
    return %0, %2, %5, %6, %8 : tensor<f32>, tensor<88xf32>, tensor<42x74x41x86xi16>, tensor<88xf32>, tensor<74x41x86xi32>
  }
}
