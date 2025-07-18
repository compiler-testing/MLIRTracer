module {
  func.func @main(%arg0: tensor<89x85x50xi8>, %arg1: tensor<37x67x90x41xf32>) -> (tensor<89x85x50xi8>, tensor<1x67x90x1xi1>, tensor<1x67x90x41xi1>) {
    %0 = tosa.clz %arg0 : (tensor<89x85x50xi8>) -> tensor<89x85x50xi8>
    %1 = tosa.bitwise_or %0, %0 : (tensor<89x85x50xi8>, tensor<89x85x50xi8>) -> tensor<89x85x50xi8>
    %2 = tosa.sigmoid %arg1 : (tensor<37x67x90x41xf32>) -> tensor<37x67x90x41xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<37x67x90x41xf32>, tensor<37x67x90x41xf32>) -> tensor<37x67x90x41xi1>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<37x67x90x41xi1>) -> tensor<1x67x90x41xi1>
    %5 = tosa.logical_or %4, %4 : (tensor<1x67x90x41xi1>, tensor<1x67x90x41xi1>) -> tensor<1x67x90x41xi1>
    %6 = tosa.reduce_all %4 {axis = 3 : i32} : (tensor<1x67x90x41xi1>) -> tensor<1x67x90x1xi1>
    %7 = tosa.logical_right_shift %5, %4 : (tensor<1x67x90x41xi1>, tensor<1x67x90x41xi1>) -> tensor<1x67x90x41xi1>
    %8 = tosa.logical_and %7, %7 : (tensor<1x67x90x41xi1>, tensor<1x67x90x41xi1>) -> tensor<1x67x90x41xi1>
    return %1, %6, %8 : tensor<89x85x50xi8>, tensor<1x67x90x1xi1>, tensor<1x67x90x41xi1>
  }
}
