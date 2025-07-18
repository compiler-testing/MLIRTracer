module {
  func.func @main(%arg0: tensor<79x70x49x65x30xf32>, %arg1: tensor<4x67x6xi32>) -> (tensor<79x70x49x65x30xf32>, tensor<79x70x49x65x30xi1>, tensor<79x70x49x65x30xf32>, tensor<79x70x49x65x30xi1>, tensor<1x1x6xi1>, tensor<1x67x6xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<79x70x49x65x30xf32>) -> tensor<79x70x49x65x30xf32>
    %1 = tosa.reduce_product %arg1 {axis = 0 : i32} : (tensor<4x67x6xi32>) -> tensor<1x67x6xi32>
    %2 = tosa.floor %0 : (tensor<79x70x49x65x30xf32>) -> tensor<79x70x49x65x30xf32>
    %3 = tosa.greater %1, %1 : (tensor<1x67x6xi32>, tensor<1x67x6xi32>) -> tensor<1x67x6xi1>
    %4 = tosa.intdiv %1, %1 : (tensor<1x67x6xi32>, tensor<1x67x6xi32>) -> tensor<1x67x6xi32>
    %5 = tosa.sigmoid %2 : (tensor<79x70x49x65x30xf32>) -> tensor<79x70x49x65x30xf32>
    %6 = tosa.equal %5, %5 : (tensor<79x70x49x65x30xf32>, tensor<79x70x49x65x30xf32>) -> tensor<79x70x49x65x30xi1>
    %7 = tosa.abs %6 : (tensor<79x70x49x65x30xi1>) -> tensor<79x70x49x65x30xi1>
    %8 = tosa.ceil %5 : (tensor<79x70x49x65x30xf32>) -> tensor<79x70x49x65x30xf32>
    %9 = tosa.add %7, %7 : (tensor<79x70x49x65x30xi1>, tensor<79x70x49x65x30xi1>) -> tensor<79x70x49x65x30xi1>
    %10 = tosa.pow %2, %2 : (tensor<79x70x49x65x30xf32>, tensor<79x70x49x65x30xf32>) -> tensor<79x70x49x65x30xf32>
    %11 = tosa.intdiv %4, %1 : (tensor<1x67x6xi32>, tensor<1x67x6xi32>) -> tensor<1x67x6xi32>
    %12 = tosa.equal %2, %0 : (tensor<79x70x49x65x30xf32>, tensor<79x70x49x65x30xf32>) -> tensor<79x70x49x65x30xi1>
    %13 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<1x67x6xi1>) -> tensor<1x1x6xi1>
    %14 = tosa.greater_equal %1, %11 : (tensor<1x67x6xi32>, tensor<1x67x6xi32>) -> tensor<1x67x6xi1>
    return %8, %9, %10, %12, %13, %14 : tensor<79x70x49x65x30xf32>, tensor<79x70x49x65x30xi1>, tensor<79x70x49x65x30xf32>, tensor<79x70x49x65x30xi1>, tensor<1x1x6xi1>, tensor<1x67x6xi1>
  }
}
