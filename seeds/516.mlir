module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<62x21x88x26xf32>) -> (tensor<i8>, tensor<62x21x88x26xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.reverse %arg2 {axis = 1 : i32} : (tensor<62x21x88x26xf32>) -> tensor<62x21x88x26xf32>
    %2 = tosa.maximum %1, %1 : (tensor<62x21x88x26xf32>, tensor<62x21x88x26xf32>) -> tensor<62x21x88x26xf32>
    %3 = tosa.equal %2, %1 : (tensor<62x21x88x26xf32>, tensor<62x21x88x26xf32>) -> tensor<62x21x88x26xi1>
    %4 = tosa.identity %3 : (tensor<62x21x88x26xi1>) -> tensor<62x21x88x26xi1>
    return %0, %4 : tensor<i8>, tensor<62x21x88x26xi1>
  }
}
