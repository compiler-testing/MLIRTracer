module {
  func.func @main(%arg0: tensor<35x41x44x30x18x32xi64>, %arg1: tensor<1x1x44x1x1x1xi64>, %arg2: tensor<51x61xi16>, %arg3: tensor<10x30x91x59xf32>) -> (tensor<10x30x91x59xf32>, tensor<35x41x44x30x18x32xi1>, tensor<51x1xi16>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<35x41x44x30x18x32xi64>, tensor<1x1x44x1x1x1xi64>) -> tensor<35x41x44x30x18x32xi1>
    %1 = tosa.reduce_sum %arg2 {axis = 1 : i32} : (tensor<51x61xi16>) -> tensor<51x1xi16>
    %2 = tosa.logical_and %0, %0 : (tensor<35x41x44x30x18x32xi1>, tensor<35x41x44x30x18x32xi1>) -> tensor<35x41x44x30x18x32xi1>
    %3 = tosa.sub %1, %1 : (tensor<51x1xi16>, tensor<51x1xi16>) -> tensor<51x1xi16>
    %4 = tosa.exp %arg3 : (tensor<10x30x91x59xf32>) -> tensor<10x30x91x59xf32>
    %5 = tosa.logical_xor %2, %0 : (tensor<35x41x44x30x18x32xi1>, tensor<35x41x44x30x18x32xi1>) -> tensor<35x41x44x30x18x32xi1>
    %6 = tosa.clz %3 : (tensor<51x1xi16>) -> tensor<51x1xi16>
    return %4, %5, %6 : tensor<10x30x91x59xf32>, tensor<35x41x44x30x18x32xi1>, tensor<51x1xi16>
  }
}
