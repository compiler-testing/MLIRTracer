module {
  func.func @main(%arg0: tensor<67x91x29x54xi32>, %arg1: tensor<1x91x29x54xi32>, %arg2: tensor<67x96x24x4xf32>, %arg3: tensor<64x66x33x21x26xi1>, %arg4: tensor<64x66x1x21x26xi1>) -> (tensor<67x91x29x108xi32>, tensor<67x96x24x4xf32>, tensor<64x66x33x21x26xi1>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<67x91x29x54xi32>, tensor<1x91x29x54xi32>) -> tensor<67x91x29x54xi32>
    %1 = tosa.concat %0, %0 {axis = 3 : i32} : (tensor<67x91x29x54xi32>, tensor<67x91x29x54xi32>) -> tensor<67x91x29x108xi32>
    %2 = tosa.log %arg2 : (tensor<67x96x24x4xf32>) -> tensor<67x96x24x4xf32>
    %3 = tosa.floor %2 : (tensor<67x96x24x4xf32>) -> tensor<67x96x24x4xf32>
    %4 = tosa.logical_and %arg3, %arg4 : (tensor<64x66x33x21x26xi1>, tensor<64x66x1x21x26xi1>) -> tensor<64x66x33x21x26xi1>
    return %1, %3, %4 : tensor<67x91x29x108xi32>, tensor<67x96x24x4xf32>, tensor<64x66x33x21x26xi1>
  }
}
