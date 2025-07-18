module {
  func.func @main(%arg0: tensor<94x43x12x83xi8>, %arg1: tensor<94x9x29xi32>, %arg2: tensor<94x1x29xi32>) -> (tensor<94x43x12x1xi8>, tensor<94x9x29xi32>) {
    %0 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<94x43x12x83xi8>) -> tensor<94x43x12x1xi8>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<94x9x29xi32>, tensor<94x1x29xi32>) -> tensor<94x9x29xi32>
    return %0, %1 : tensor<94x43x12x1xi8>, tensor<94x9x29xi32>
  }
}
