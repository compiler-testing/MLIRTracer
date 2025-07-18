module {
  func.func @main(%arg0: tensor<77x39x91x9x38x64xi8>, %arg1: tensor<77x39x91x1x1x64xi8>, %arg2: tensor<5x14xi64>, %arg3: tensor<97x53x41x90xi1>, %arg4: tensor<97x53x1x90xi1>, %arg5: tensor<21xf32>) -> (tensor<77x39x91x9x38x64xi8>, tensor<5x1xi64>, tensor<97x53x41x90xi1>, tensor<21xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<77x39x91x9x38x64xi8>, tensor<77x39x91x1x1x64xi8>) -> tensor<77x39x91x9x38x64xi8>
    %1 = tosa.reduce_product %arg2 {axis = 1 : i32} : (tensor<5x14xi64>) -> tensor<5x1xi64>
    %2 = tosa.logical_and %arg3, %arg4 : (tensor<97x53x41x90xi1>, tensor<97x53x1x90xi1>) -> tensor<97x53x41x90xi1>
    %3 = tosa.sub %2, %2 : (tensor<97x53x41x90xi1>, tensor<97x53x41x90xi1>) -> tensor<97x53x41x90xi1>
    %4 = tosa.sigmoid %arg5 : (tensor<21xf32>) -> tensor<21xf32>
    %5 = tosa.tanh %4 : (tensor<21xf32>) -> tensor<21xf32>
    return %0, %1, %3, %5 : tensor<77x39x91x9x38x64xi8>, tensor<5x1xi64>, tensor<97x53x41x90xi1>, tensor<21xf32>
  }
}
