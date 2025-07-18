module {
  func.func @main(%arg0: tensor<42x40x17x82xf32>, %arg1: tensor<56x43x8x45xi1>, %arg2: tensor<1x43x1x45xi1>) -> (tensor<42x40x17x82xf32>, tensor<56x43x8x45xi1>) {
    %0 = tosa.reverse %arg0 {axis = 3 : i32} : (tensor<42x40x17x82xf32>) -> tensor<42x40x17x82xf32>
    %1 = tosa.reciprocal %0 : (tensor<42x40x17x82xf32>) -> tensor<42x40x17x82xf32>
    %2 = tosa.abs %1 : (tensor<42x40x17x82xf32>) -> tensor<42x40x17x82xf32>
    %3 = tosa.logical_xor %arg1, %arg2 : (tensor<56x43x8x45xi1>, tensor<1x43x1x45xi1>) -> tensor<56x43x8x45xi1>
    return %2, %3 : tensor<42x40x17x82xf32>, tensor<56x43x8x45xi1>
  }
}
