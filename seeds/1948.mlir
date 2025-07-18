module {
  func.func @main(%arg0: tensor<11xi1>, %arg1: tensor<1xi1>, %arg2: tensor<68x79x90x31x8xi32>, %arg3: tensor<68x79x90x31x1xi32>) -> (tensor<11x2xi1>, tensor<68x79x90x31x8xi32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<11xi1>, tensor<1xi1>) -> tensor<11xi1>
    %r_1 = tosa.const_shape {values = dense<[ 11, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<11xi1>, !tosa.shape<2>) -> tensor<11x1xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<11x1xi1>, tensor<11x1xi1>) -> tensor<11x1xi1>
    %3 = tosa.intdiv %arg2, %arg3 : (tensor<68x79x90x31x8xi32>, tensor<68x79x90x31x1xi32>) -> tensor<68x79x90x31x8xi32>
    %4 = tosa.logical_or %2, %1 : (tensor<11x1xi1>, tensor<11x1xi1>) -> tensor<11x1xi1>
    %5 = tosa.concat %4, %4 {axis = 1 : i32} : (tensor<11x1xi1>, tensor<11x1xi1>) -> tensor<11x2xi1>
    %6 = tosa.add %3, %3 : (tensor<68x79x90x31x8xi32>, tensor<68x79x90x31x8xi32>) -> tensor<68x79x90x31x8xi32>
    %7 = tosa.maximum %6, %6 : (tensor<68x79x90x31x8xi32>, tensor<68x79x90x31x8xi32>) -> tensor<68x79x90x31x8xi32>
    return %5, %7 : tensor<11x2xi1>, tensor<68x79x90x31x8xi32>
  }
}
