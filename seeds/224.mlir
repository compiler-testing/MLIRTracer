module {
  func.func @main(%arg0: tensor<71x29x21x62x12xi1>, %arg1: tensor<24x29x14xf32>, %arg2: tensor<24x29x1xf32>, %arg3: tensor<82x99x79xi1>) -> (tensor<71x29x21x62x12xi1>, tensor<24x29x14xf32>, tensor<1x99x79xi1>, tensor<24x29x14xf32>) {
    %0 = tosa.clz %arg0 : (tensor<71x29x21x62x12xi1>) -> tensor<71x29x21x62x12xi1>
    %1 = tosa.pow %arg1, %arg2 : (tensor<24x29x14xf32>, tensor<24x29x1xf32>) -> tensor<24x29x14xf32>
    %2 = tosa.sigmoid %1 : (tensor<24x29x14xf32>) -> tensor<24x29x14xf32>
    %3 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<82x99x79xi1>) -> tensor<1x99x79xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<1x99x79xi1>, tensor<1x99x79xi1>) -> tensor<1x99x79xi1>
    %5 = tosa.bitwise_and %4, %3 : (tensor<1x99x79xi1>, tensor<1x99x79xi1>) -> tensor<1x99x79xi1>
    %6 = tosa.sigmoid %1 : (tensor<24x29x14xf32>) -> tensor<24x29x14xf32>
    return %0, %2, %5, %6 : tensor<71x29x21x62x12xi1>, tensor<24x29x14xf32>, tensor<1x99x79xi1>, tensor<24x29x14xf32>
  }
}
