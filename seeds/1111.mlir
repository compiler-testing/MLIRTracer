module {
  func.func @main(%arg0: tensor<80x70x29x89x24xi8>, %arg1: tensor<80x1x29x89x24xi8>, %arg2: tensor<35x37xi32>, %arg3: tensor<22xi1>) -> (tensor<80x70x29x89x24xi8>, tensor<1x37xi32>, tensor<1xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<80x70x29x89x24xi8>, tensor<80x1x29x89x24xi8>) -> tensor<80x70x29x89x24xi8>
    %1 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<35x37xi32>) -> tensor<1x37xi32>
    %2 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<22xi1>) -> tensor<1xi1>
    return %0, %1, %2 : tensor<80x70x29x89x24xi8>, tensor<1x37xi32>, tensor<1xi1>
  }
}
