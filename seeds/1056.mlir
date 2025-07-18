module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<67xi1>, %arg2: tensor<52x23xi32>, %arg3: tensor<1x1xi32>, %arg4: tensor<70x18x57x46x58x22xf32>) -> (tensor<1xi1>, tensor<i1>, tensor<70x18x57x46x58x22xf32>, tensor<52x23xi32>, tensor<70x18x57x46x58x22xf32>) {
    %0 = tosa.clz %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.bitwise_not %0 : (tensor<i1>) -> tensor<i1>
    %2 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<67xi1>) -> tensor<1xi1>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.intdiv %arg2, %arg3 : (tensor<52x23xi32>, tensor<1x1xi32>) -> tensor<52x23xi32>
    %5 = tosa.ceil %arg4 : (tensor<70x18x57x46x58x22xf32>) -> tensor<70x18x57x46x58x22xf32>
    %6 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.rsqrt %5 : (tensor<70x18x57x46x58x22xf32>) -> tensor<70x18x57x46x58x22xf32>
    %8 = tosa.maximum %5, %7 : (tensor<70x18x57x46x58x22xf32>, tensor<70x18x57x46x58x22xf32>) -> tensor<70x18x57x46x58x22xf32>
    %9 = tosa.minimum %8, %5 : (tensor<70x18x57x46x58x22xf32>, tensor<70x18x57x46x58x22xf32>) -> tensor<70x18x57x46x58x22xf32>
    %10 = tosa.logical_xor %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.maximum %8, %9 : (tensor<70x18x57x46x58x22xf32>, tensor<70x18x57x46x58x22xf32>) -> tensor<70x18x57x46x58x22xf32>
    %12 = tosa.floor %11 : (tensor<70x18x57x46x58x22xf32>) -> tensor<70x18x57x46x58x22xf32>
    %13 = tosa.bitwise_xor %4, %4 : (tensor<52x23xi32>, tensor<52x23xi32>) -> tensor<52x23xi32>
    %14 = tosa.exp %11 : (tensor<70x18x57x46x58x22xf32>) -> tensor<70x18x57x46x58x22xf32>
    return %6, %10, %12, %13, %14 : tensor<1xi1>, tensor<i1>, tensor<70x18x57x46x58x22xf32>, tensor<52x23xi32>, tensor<70x18x57x46x58x22xf32>
  }
}
