module {
  func.func @main(%arg0: tensor<45x51xi16>, %arg1: tensor<10x78x28x60x70xf32>, %arg2: tensor<75x91x53xi1>) -> (tensor<135x51xi16>, tensor<75x91x1xi1>, tensor<10x78x28x60x70xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<45x51xi16>, !tosa.shape<2>) -> tensor<135x51xi16>
    %1 = tosa.tanh %arg1 : (tensor<10x78x28x60x70xf32>) -> tensor<10x78x28x60x70xf32>
    %2 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<135x51xi16>, tensor<135x51xi16>) -> tensor<135x51xi16>
    %3 = tosa.reduce_any %arg2 {axis = 2 : i32} : (tensor<75x91x53xi1>) -> tensor<75x91x1xi1>
    %4 = tosa.ceil %1 : (tensor<10x78x28x60x70xf32>) -> tensor<10x78x28x60x70xf32>
    %5 = tosa.abs %4 : (tensor<10x78x28x60x70xf32>) -> tensor<10x78x28x60x70xf32>
    return %2, %3, %5 : tensor<135x51xi16>, tensor<75x91x1xi1>, tensor<10x78x28x60x70xf32>
  }
}
