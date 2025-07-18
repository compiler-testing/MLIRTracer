module {
  func.func @main(%arg0: tensor<6xf32>, %arg1: tensor<91xi16>, %arg2: tensor<91xi16>) -> (tensor<1xi16>, tensor<6xi1>) {
    %0 = tosa.exp %arg0 : (tensor<6xf32>) -> tensor<6xf32>
    %1 = tosa.bitwise_or %arg1, %arg2 : (tensor<91xi16>, tensor<91xi16>) -> tensor<91xi16>
    %2 = tosa.greater_equal %0, %0 : (tensor<6xf32>, tensor<6xf32>) -> tensor<6xi1>
    %3 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<91xi16>) -> tensor<1xi16>
    %4 = tosa.bitwise_xor %2, %2 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %5 = tosa.logical_or %4, %2 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %6 = tosa.logical_and %5, %4 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    %7 = tosa.sub %6, %6 : (tensor<6xi1>, tensor<6xi1>) -> tensor<6xi1>
    return %3, %7 : tensor<1xi16>, tensor<6xi1>
  }
}
