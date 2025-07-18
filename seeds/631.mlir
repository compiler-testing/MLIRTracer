module {
  func.func @main(%arg0: tensor<35x83x66x17xi32>, %arg1: tensor<27x18x62x44xi1>) -> (tensor<1x1x62x44xi1>, tensor<35x1x1x17xi32>) {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<35x83x66x17xi32>) -> tensor<35x83x66x17xi32>
    %1 = tosa.clz %0 : (tensor<35x83x66x17xi32>) -> tensor<35x83x66x17xi32>
    %2 = tosa.reduce_product %1 {axis = 1 : i32} : (tensor<35x83x66x17xi32>) -> tensor<35x1x66x17xi32>
    %3 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<27x18x62x44xi1>) -> tensor<1x18x62x44xi1>
    %4 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<1x18x62x44xi1>) -> tensor<1x1x62x44xi1>
    %5 = tosa.reduce_product %2 {axis = 2 : i32} : (tensor<35x1x66x17xi32>) -> tensor<35x1x1x17xi32>
    %6 = tosa.bitwise_not %5 : (tensor<35x1x1x17xi32>) -> tensor<35x1x1x17xi32>
    return %4, %6 : tensor<1x1x62x44xi1>, tensor<35x1x1x17xi32>
  }
}
