module {
  func.func @main(%arg0: tensor<51x41x61x6xi32>, %arg1: tensor<1x1x1x6xi32>, %arg2: tensor<47x99xi1>, %arg3: tensor<69x40x13x45x69x12xf32>) -> (tensor<51x41x61x6xi32>, tensor<51x41x61x6xi32>, tensor<1x1xi1>, tensor<69x40x13x45x69x12xf32>, tensor<1x1xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<51x41x61x6xi32>, tensor<1x1x1x6xi32>) -> tensor<51x41x61x6xi32>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<47x99xi1>) -> tensor<1x99xi1>
    %2 = tosa.add %1, %1 : (tensor<1x99xi1>, tensor<1x99xi1>) -> tensor<1x99xi1>
    %3 = tosa.minimum %0, %0 : (tensor<51x41x61x6xi32>, tensor<51x41x61x6xi32>) -> tensor<51x41x61x6xi32>
    %4 = tosa.reduce_all %2 {axis = 1 : i32} : (tensor<1x99xi1>) -> tensor<1x1xi1>
    %5 = tosa.bitwise_xor %2, %1 : (tensor<1x99xi1>, tensor<1x99xi1>) -> tensor<1x99xi1>
    %6 = tosa.bitwise_xor %5, %1 : (tensor<1x99xi1>, tensor<1x99xi1>) -> tensor<1x99xi1>
    %7 = tosa.intdiv %0, %0 : (tensor<51x41x61x6xi32>, tensor<51x41x61x6xi32>) -> tensor<51x41x61x6xi32>
    %8 = tosa.maximum %3, %3 : (tensor<51x41x61x6xi32>, tensor<51x41x61x6xi32>) -> tensor<51x41x61x6xi32>
    %9 = tosa.logical_left_shift %4, %4 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %10 = tosa.abs %6 : (tensor<1x99xi1>) -> tensor<1x99xi1>
    %11 = tosa.reciprocal %arg3 : (tensor<69x40x13x45x69x12xf32>) -> tensor<69x40x13x45x69x12xf32>
    %12 = tosa.reduce_sum %10 {axis = 1 : i32} : (tensor<1x99xi1>) -> tensor<1x1xi1>
    %13 = tosa.abs %12 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    return %7, %8, %9, %11, %13 : tensor<51x41x61x6xi32>, tensor<51x41x61x6xi32>, tensor<1x1xi1>, tensor<69x40x13x45x69x12xf32>, tensor<1x1xi1>
  }
}
