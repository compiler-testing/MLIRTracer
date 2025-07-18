module {
  func.func @main(%arg0: tensor<77xi16>, %arg1: tensor<77xi16>, %arg2: tensor<61x44x22x14x48x79xf32>) -> (tensor<61x44x22x14x48x79xf32>, tensor<1xi16>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<77xi16>, tensor<77xi16>) -> tensor<77xi16>
    %1 = tosa.exp %arg2 : (tensor<61x44x22x14x48x79xf32>) -> tensor<61x44x22x14x48x79xf32>
    %2 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<77xi16>) -> tensor<1xi16>
    return %1, %2 : tensor<61x44x22x14x48x79xf32>, tensor<1xi16>
  }
}
