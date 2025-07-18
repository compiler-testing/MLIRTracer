module {
  func.func @main(%arg0: tensor<6x85x21xi8>, %arg1: tensor<89x65x68x25x30x72xi1>, %arg2: tensor<31x7x66xi1>, %arg3: tensor<78x81x94x91xf32>) -> (tensor<6x85x21xi8>, tensor<78x81x94x91xf32>, tensor<89x65x68x25x30x72xi1>, tensor<31x7x1xi1>, tensor<31x1xi1>, tensor<31x1xi32>, tensor<78x81x1x91xf32>, tensor<78x81x94x91xf32>, tensor<31x1xi1>) {
    %0 = tosa.abs %arg0 : (tensor<6x85x21xi8>) -> tensor<6x85x21xi8>
    %1 = tosa.abs %0 : (tensor<6x85x21xi8>) -> tensor<6x85x21xi8>
    %2 = tosa.minimum %1, %1 : (tensor<6x85x21xi8>, tensor<6x85x21xi8>) -> tensor<6x85x21xi8>
    %3 = tosa.logical_not %arg1 : (tensor<89x65x68x25x30x72xi1>) -> tensor<89x65x68x25x30x72xi1>
    %4 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<31x7x66xi1>) -> tensor<31x7x1xi1>
    %5 = tosa.ceil %arg3 : (tensor<78x81x94x91xf32>) -> tensor<78x81x94x91xf32>
    %6 = tosa.ceil %5 : (tensor<78x81x94x91xf32>) -> tensor<78x81x94x91xf32>
    %7 = tosa.logical_and %3, %3 : (tensor<89x65x68x25x30x72xi1>, tensor<89x65x68x25x30x72xi1>) -> tensor<89x65x68x25x30x72xi1>
    %8 = tosa.argmax %4 {axis = 1 : i32} : (tensor<31x7x1xi1>) -> tensor<31x1xi32>
    %9 = tosa.bitwise_or %4, %4 : (tensor<31x7x1xi1>, tensor<31x7x1xi1>) -> tensor<31x7x1xi1>
    %10 = tosa.logical_left_shift %8, %8 : (tensor<31x1xi32>, tensor<31x1xi32>) -> tensor<31x1xi32>
    %11 = tosa.logical_xor %9, %9 : (tensor<31x7x1xi1>, tensor<31x7x1xi1>) -> tensor<31x7x1xi1>
    %12 = tosa.reverse %8 {axis = 0 : i32} : (tensor<31x1xi32>) -> tensor<31x1xi32>
    %13 = tosa.minimum %12, %10 : (tensor<31x1xi32>, tensor<31x1xi32>) -> tensor<31x1xi32>
    %14 = tosa.equal %13, %8 : (tensor<31x1xi32>, tensor<31x1xi32>) -> tensor<31x1xi1>
    %15 = tosa.reciprocal %5 : (tensor<78x81x94x91xf32>) -> tensor<78x81x94x91xf32>
    %16 = tosa.reduce_sum %12 {axis = 1 : i32} : (tensor<31x1xi32>) -> tensor<31x1xi32>
    %17 = tosa.reduce_sum %5 {axis = 2 : i32} : (tensor<78x81x94x91xf32>) -> tensor<78x81x1x91xf32>
    %18 = tosa.tanh %15 : (tensor<78x81x94x91xf32>) -> tensor<78x81x94x91xf32>
    %19 = tosa.greater %12, %13 : (tensor<31x1xi32>, tensor<31x1xi32>) -> tensor<31x1xi1>
    return %2, %6, %7, %11, %14, %16, %17, %18, %19 : tensor<6x85x21xi8>, tensor<78x81x94x91xf32>, tensor<89x65x68x25x30x72xi1>, tensor<31x7x1xi1>, tensor<31x1xi1>, tensor<31x1xi32>, tensor<78x81x1x91xf32>, tensor<78x81x94x91xf32>, tensor<31x1xi1>
  }
}
