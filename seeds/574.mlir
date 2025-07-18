module {
  func.func @main(%arg0: tensor<18x89x78x26xf32>) -> (tensor<6942xi1>, tensor<18x89x78x26xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<18x89x78x26xf32>) -> tensor<18x89x78x26xf32>
    %1 = tosa.rsqrt %0 : (tensor<18x89x78x26xf32>) -> tensor<18x89x78x26xf32>
    %2 = tosa.equal %1, %0 : (tensor<18x89x78x26xf32>, tensor<18x89x78x26xf32>) -> tensor<18x89x78x26xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<18x89x78x26xi1>, tensor<18x89x78x26xi1>) -> tensor<18x89x78x26xi1>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<18x89x78x26xi1>) -> tensor<1x89x78x26xi1>
    %5 = tosa.reduce_product %4 {axis = 3 : i32} : (tensor<1x89x78x26xi1>) -> tensor<1x89x78x1xi1>
    %r_6 = tosa.const_shape {values = dense<[ 6942 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %5, %r_6 : (tensor<1x89x78x1xi1>, !tosa.shape<1>) -> tensor<6942xi1>
    %7 = tosa.floor %1 : (tensor<18x89x78x26xf32>) -> tensor<18x89x78x26xf32>
    return %6, %7 : tensor<6942xi1>, tensor<18x89x78x26xf32>
  }
}
