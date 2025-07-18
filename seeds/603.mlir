module {
  func.func @main(%arg0: tensor<83xi1>, %arg1: tensor<83xi1>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<83x86x70x29xf32>) -> (tensor<83xi1>, tensor<i32>, tensor<83x86x70x29xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<83xi1>, tensor<83xi1>) -> tensor<83xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<83xi1>, tensor<83xi1>) -> tensor<83xi1>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.rsqrt %arg4 : (tensor<83x86x70x29xf32>) -> tensor<83x86x70x29xf32>
    %4 = tosa.equal %3, %3 : (tensor<83x86x70x29xf32>, tensor<83x86x70x29xf32>) -> tensor<83x86x70x29xi1>
    return %1, %2, %4 : tensor<83xi1>, tensor<i32>, tensor<83x86x70x29xi1>
  }
}
