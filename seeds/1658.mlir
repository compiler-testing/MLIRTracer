module {
  func.func @main(%arg0: tensor<59x91x57x43x53xi16>, %arg1: tensor<1x1x57x43x1xi16>, %arg2: tensor<49xf32>, %arg3: tensor<1xf32>, %arg4: tensor<61x5xi8>, %arg5: tensor<1x5xi8>) -> (tensor<59x91x57x43x53xi16>, tensor<49xi1>, tensor<61x5xi8>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<59x91x57x43x53xi16>, tensor<1x1x57x43x1xi16>) -> tensor<59x91x57x43x53xi16>
    %1 = tosa.equal %arg2, %arg3 : (tensor<49xf32>, tensor<1xf32>) -> tensor<49xi1>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<49xi1>, tensor<49xi1>) -> tensor<49xi1>
    %3 = tosa.sub %2, %2 : (tensor<49xi1>, tensor<49xi1>) -> tensor<49xi1>
    %4 = tosa.minimum %arg4, %arg5 : (tensor<61x5xi8>, tensor<1x5xi8>) -> tensor<61x5xi8>
    return %0, %3, %4 : tensor<59x91x57x43x53xi16>, tensor<49xi1>, tensor<61x5xi8>
  }
}
