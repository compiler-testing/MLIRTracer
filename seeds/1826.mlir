module {
  func.func @main(%arg0: tensor<38x36x97xi1>, %arg1: tensor<88xi8>, %arg2: tensor<88xi8>) -> (tensor<2x36x1xi1>, tensor<88xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 2 : i32} : (tensor<38x36x97xi1>) -> tensor<38x36x1xi1>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<38x36x1xi1>) -> tensor<1x36x1xi1>
    %2 = tosa.clz %1 : (tensor<1x36x1xi1>) -> tensor<1x36x1xi1>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<1x36x1xi1>, tensor<1x36x1xi1>) -> tensor<2x36x1xi1>
    %4 = tosa.equal %arg1, %arg2 : (tensor<88xi8>, tensor<88xi8>) -> tensor<88xi1>
    return %3, %4 : tensor<2x36x1xi1>, tensor<88xi1>
  }
}
