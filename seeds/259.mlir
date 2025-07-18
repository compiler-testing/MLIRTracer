module {
  func.func @main(%arg0: tensor<36x64x77x82x24xf32>, %arg1: tensor<66xi32>, %arg2: tensor<7x8x16xi1>) -> (tensor<i32>, tensor<7x8x1xi1>, tensor<36x64x77x82x24xf32>, tensor<36x64x77x82x24xf32>, tensor<7x1x1xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<36x64x77x82x24xf32>) -> tensor<36x64x77x82x24xf32>
    %1 = tosa.argmax %arg1 {axis = 0 : i32} : (tensor<66xi32>) -> tensor<i32>
    %2 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<7x8x16xi1>) -> tensor<7x8x1xi1>
    %3 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<7x8x1xi1>) -> tensor<7x1x1xi1>
    %4 = tosa.identity %2 : (tensor<7x8x1xi1>) -> tensor<7x8x1xi1>
    %5 = tosa.reciprocal %0 : (tensor<36x64x77x82x24xf32>) -> tensor<36x64x77x82x24xf32>
    %6 = tosa.rsqrt %0 : (tensor<36x64x77x82x24xf32>) -> tensor<36x64x77x82x24xf32>
    %7 = tosa.reverse %3 {axis = 0 : i32} : (tensor<7x1x1xi1>) -> tensor<7x1x1xi1>
    return %1, %4, %5, %6, %7 : tensor<i32>, tensor<7x8x1xi1>, tensor<36x64x77x82x24xf32>, tensor<36x64x77x82x24xf32>, tensor<7x1x1xi1>
  }
}
