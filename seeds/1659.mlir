module {
  func.func @main(%arg0: tensor<15x97xf32>, %arg1: tensor<24x73x35x5xi1>, %arg2: tensor<41x66xi32>, %arg3: tensor<1x1xi32>) -> (tensor<24x1x35x5xi1>, tensor<15x97xi1>, tensor<41x66xi32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<15x97xf32>) -> tensor<15x97xf32>
    %1 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<24x73x35x5xi1>) -> tensor<24x1x35x5xi1>
    %2 = tosa.greater %0, %0 : (tensor<15x97xf32>, tensor<15x97xf32>) -> tensor<15x97xi1>
    %3 = tosa.intdiv %arg2, %arg3 : (tensor<41x66xi32>, tensor<1x1xi32>) -> tensor<41x66xi32>
    return %1, %2, %3 : tensor<24x1x35x5xi1>, tensor<15x97xi1>, tensor<41x66xi32>
  }
}
