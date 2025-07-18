module {
  func.func @main(%arg0: tensor<28x89x51x38x41x43xi8>, %arg1: tensor<28x89x51x38x1x1xi8>, %arg2: tensor<76xi64>) -> (tensor<28x89x51x38x41x43xi1>, tensor<1xi64>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<28x89x51x38x41x43xi8>, tensor<28x89x51x38x1x1xi8>) -> tensor<28x89x51x38x41x43xi1>
    %1 = tosa.abs %0 : (tensor<28x89x51x38x41x43xi1>) -> tensor<28x89x51x38x41x43xi1>
    %2 = tosa.add %1, %0 : (tensor<28x89x51x38x41x43xi1>, tensor<28x89x51x38x41x43xi1>) -> tensor<28x89x51x38x41x43xi1>
    %3 = tosa.reduce_min %arg2 {axis = 0 : i32} : (tensor<76xi64>) -> tensor<1xi64>
    return %2, %3 : tensor<28x89x51x38x41x43xi1>, tensor<1xi64>
  }
}
