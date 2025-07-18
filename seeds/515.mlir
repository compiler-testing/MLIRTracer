module {
  func.func @main(%arg0: tensor<63x28xf32>, %arg1: tensor<51x94x7x72xi16>, %arg2: tensor<51x1x7x72xi16>) -> (tensor<51x94x7x72xi16>, tensor<63x28xi1>) {
    %0 = tosa.exp %arg0 : (tensor<63x28xf32>) -> tensor<63x28xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<51x94x7x72xi16>, tensor<51x1x7x72xi16>) -> tensor<51x94x7x72xi16>
    %2 = tosa.equal %0, %0 : (tensor<63x28xf32>, tensor<63x28xf32>) -> tensor<63x28xi1>
    return %1, %2 : tensor<51x94x7x72xi16>, tensor<63x28xi1>
  }
}
