module {
  func.func @main(%arg0: tensor<32x1x4x70x31x66xi8>, %arg1: tensor<32x1x1x70x1x1xi8>, %arg2: tensor<83x66x77x29xi32>, %arg3: tensor<83x66x77x29xi32>, %arg4: tensor<97x87x56x33xi1>, %arg5: tensor<1x87x56x1xi1>, %arg6: tensor<77x40x82x56x95xf32>) -> (tensor<32x1x4x70x31x66xi8>, tensor<83x77x29xi32>, tensor<83x66x77x29xi32>, tensor<97x87x56x33xi1>, tensor<77x40x82x56x95xf32>, tensor<97x87x56x33xi1>, tensor<97x1x56x33xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<32x1x4x70x31x66xi8>, tensor<32x1x1x70x1x1xi8>) -> tensor<32x1x4x70x31x66xi8>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<83x66x77x29xi32>, tensor<83x66x77x29xi32>) -> tensor<83x66x77x29xi32>
    %2 = tosa.maximum %0, %0 : (tensor<32x1x4x70x31x66xi8>, tensor<32x1x4x70x31x66xi8>) -> tensor<32x1x4x70x31x66xi8>
    %3 = tosa.argmax %1 {axis = 1 : i32} : (tensor<83x66x77x29xi32>) -> tensor<83x77x29xi32>
    %4 = tosa.bitwise_or %1, %1 : (tensor<83x66x77x29xi32>, tensor<83x66x77x29xi32>) -> tensor<83x66x77x29xi32>
    %5 = tosa.logical_or %arg4, %arg5 : (tensor<97x87x56x33xi1>, tensor<1x87x56x1xi1>) -> tensor<97x87x56x33xi1>
    %6 = tosa.bitwise_and %5, %5 : (tensor<97x87x56x33xi1>, tensor<97x87x56x33xi1>) -> tensor<97x87x56x33xi1>
    %7 = tosa.ceil %arg6 : (tensor<77x40x82x56x95xf32>) -> tensor<77x40x82x56x95xf32>
    %8 = tosa.bitwise_or %5, %5 : (tensor<97x87x56x33xi1>, tensor<97x87x56x33xi1>) -> tensor<97x87x56x33xi1>
    %9 = tosa.reduce_all %5 {axis = 1 : i32} : (tensor<97x87x56x33xi1>) -> tensor<97x1x56x33xi1>
    return %2, %3, %4, %6, %7, %8, %9 : tensor<32x1x4x70x31x66xi8>, tensor<83x77x29xi32>, tensor<83x66x77x29xi32>, tensor<97x87x56x33xi1>, tensor<77x40x82x56x95xf32>, tensor<97x87x56x33xi1>, tensor<97x1x56x33xi1>
  }
}
