module {
  func.func @main(%arg0: tensor<20x28x32xi1>, %arg1: tensor<62x7x74x30xf32>, %arg2: tensor<1x1x1x30xf32>, %arg3: tensor<30x9x84x48xf32>) -> (tensor<62x7x74x30xi1>, tensor<20x1x1xi1>, tensor<30x9x84x48xf32>, tensor<30x9x84x48xf32>) {
    %0 = tosa.clz %arg0 : (tensor<20x28x32xi1>) -> tensor<20x28x32xi1>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<20x28x32xi1>) -> tensor<20x1x32xi1>
    %2 = tosa.reduce_max %1 {axis = 1 : i32} : (tensor<20x1x32xi1>) -> tensor<20x1x32xi1>
    %3 = tosa.bitwise_not %2 : (tensor<20x1x32xi1>) -> tensor<20x1x32xi1>
    %4 = tosa.equal %arg1, %arg2 : (tensor<62x7x74x30xf32>, tensor<1x1x1x30xf32>) -> tensor<62x7x74x30xi1>
    %5 = tosa.reciprocal %arg3 : (tensor<30x9x84x48xf32>) -> tensor<30x9x84x48xf32>
    %6 = tosa.reduce_all %3 {axis = 2 : i32} : (tensor<20x1x32xi1>) -> tensor<20x1x1xi1>
    %7 = tosa.exp %5 : (tensor<30x9x84x48xf32>) -> tensor<30x9x84x48xf32>
    %8 = tosa.pow %5, %5 : (tensor<30x9x84x48xf32>, tensor<30x9x84x48xf32>) -> tensor<30x9x84x48xf32>
    %9 = tosa.reciprocal %8 : (tensor<30x9x84x48xf32>) -> tensor<30x9x84x48xf32>
    return %4, %6, %7, %9 : tensor<62x7x74x30xi1>, tensor<20x1x1xi1>, tensor<30x9x84x48xf32>, tensor<30x9x84x48xf32>
  }
}
