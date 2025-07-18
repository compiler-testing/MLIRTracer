module {
  func.func @main(%arg0: tensor<44x32x15xi32>, %arg1: tensor<1x32x1xi32>, %arg2: tensor<83x84x30xi1>, %arg3: tensor<1x84x1xi1>, %arg4: tensor<82x61x98x31x34xf32>) -> (tensor<44x32x15xi32>, tensor<83x1x30xi1>, tensor<82x61x98x31x34xf32>, tensor<83x84x30xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<44x32x15xi32>, tensor<1x32x1xi32>) -> tensor<44x32x15xi32>
    %1 = tosa.add %0, %0 : (tensor<44x32x15xi32>, tensor<44x32x15xi32>) -> tensor<44x32x15xi32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<83x84x30xi1>, tensor<1x84x1xi1>) -> tensor<83x84x30xi1>
    %3 = tosa.reduce_all %2 {axis = 1 : i32} : (tensor<83x84x30xi1>) -> tensor<83x1x30xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<83x1x30xi1>, tensor<83x1x30xi1>) -> tensor<83x1x30xi1>
    %5 = tosa.exp %arg4 : (tensor<82x61x98x31x34xf32>) -> tensor<82x61x98x31x34xf32>
    %6 = tosa.sub %2, %2 : (tensor<83x84x30xi1>, tensor<83x84x30xi1>) -> tensor<83x84x30xi1>
    return %1, %4, %5, %6 : tensor<44x32x15xi32>, tensor<83x1x30xi1>, tensor<82x61x98x31x34xf32>, tensor<83x84x30xi1>
  }
}
