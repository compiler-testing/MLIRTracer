module {
  func.func @main(%arg0: tensor<76x51x30xi1>, %arg1: tensor<34xf32>) -> (tensor<76x51x30xi1>, tensor<34xf32>, tensor<76x1x1xi1>) {
    %0 = tosa.abs %arg0 : (tensor<76x51x30xi1>) -> tensor<76x51x30xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<76x51x30xi1>, tensor<76x51x30xi1>) -> tensor<76x51x30xi1>
    %2 = tosa.exp %arg1 : (tensor<34xf32>) -> tensor<34xf32>
    %3 = tosa.bitwise_or %1, %0 : (tensor<76x51x30xi1>, tensor<76x51x30xi1>) -> tensor<76x51x30xi1>
    %4 = tosa.exp %2 : (tensor<34xf32>) -> tensor<34xf32>
    %5 = tosa.reduce_any %0 {axis = 2 : i32} : (tensor<76x51x30xi1>) -> tensor<76x51x1xi1>
    %6 = tosa.reduce_all %5 {axis = 1 : i32} : (tensor<76x51x1xi1>) -> tensor<76x1x1xi1>
    return %3, %4, %6 : tensor<76x51x30xi1>, tensor<34xf32>, tensor<76x1x1xi1>
  }
}
