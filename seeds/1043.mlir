module {
  func.func @main(%arg0: tensor<15x27xf32>, %arg1: tensor<49x78xi16>, %arg2: tensor<49x1xi16>, %arg3: tensor<78x31x42x86xi1>) -> (tensor<15x27xf32>, tensor<49xi32>, tensor<1x1x42x86xi1>, tensor<1x1x42x86xi1>, tensor<15x27xf32>) {
    %0 = tosa.exp %arg0 : (tensor<15x27xf32>) -> tensor<15x27xf32>
    %1 = tosa.bitwise_or %arg1, %arg2 : (tensor<49x78xi16>, tensor<49x1xi16>) -> tensor<49x78xi16>
    %2 = tosa.bitwise_or %1, %1 : (tensor<49x78xi16>, tensor<49x78xi16>) -> tensor<49x78xi16>
    %3 = tosa.floor %0 : (tensor<15x27xf32>) -> tensor<15x27xf32>
    %4 = tosa.argmax %2 {axis = 1 : i32} : (tensor<49x78xi16>) -> tensor<49xi32>
    %5 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<78x31x42x86xi1>) -> tensor<1x31x42x86xi1>
    %6 = tosa.log %0 : (tensor<15x27xf32>) -> tensor<15x27xf32>
    %7 = tosa.reduce_all %5 {axis = 1 : i32} : (tensor<1x31x42x86xi1>) -> tensor<1x1x42x86xi1>
    %8 = tosa.reduce_sum %5 {axis = 0 : i32} : (tensor<1x31x42x86xi1>) -> tensor<1x31x42x86xi1>
    %9 = tosa.reduce_sum %8 {axis = 1 : i32} : (tensor<1x31x42x86xi1>) -> tensor<1x1x42x86xi1>
    %10 = tosa.ceil %6 : (tensor<15x27xf32>) -> tensor<15x27xf32>
    %11 = tosa.logical_not %9 : (tensor<1x1x42x86xi1>) -> tensor<1x1x42x86xi1>
    %12 = tosa.abs %7 : (tensor<1x1x42x86xi1>) -> tensor<1x1x42x86xi1>
    %13 = tosa.exp %10 : (tensor<15x27xf32>) -> tensor<15x27xf32>
    return %3, %4, %11, %12, %13 : tensor<15x27xf32>, tensor<49xi32>, tensor<1x1x42x86xi1>, tensor<1x1x42x86xi1>, tensor<15x27xf32>
  }
}
