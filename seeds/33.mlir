module {
  func.func @main(%arg0: tensor<51xf32>) -> (tensor<51xi1>, tensor<51xf32>, tensor<1xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<51xf32>) -> tensor<51xf32>
    %1 = tosa.greater_equal %0, %0 : (tensor<51xf32>, tensor<51xf32>) -> tensor<51xi1>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<51xi1>) -> tensor<1xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.maximum %0, %0 : (tensor<51xf32>, tensor<51xf32>) -> tensor<51xf32>
    %5 = tosa.equal %0, %4 : (tensor<51xf32>, tensor<51xf32>) -> tensor<51xi1>
    %6 = tosa.rsqrt %0 : (tensor<51xf32>) -> tensor<51xf32>
    %7 = tosa.sub %3, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %5, %6, %7 : tensor<51xi1>, tensor<51xf32>, tensor<1xi1>
  }
}
