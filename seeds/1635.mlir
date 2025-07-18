module {
  func.func @main(%arg0: tensor<68x79x68x70xi1>, %arg1: tensor<68x79x63x70xi1>, %arg2: tensor<9x29x62x10xf32>, %arg3: tensor<1x1x1x1xf32>) -> (tensor<204x237x393x140xi1>, tensor<9x29x62x10xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<68x79x68x70xi1>, tensor<68x79x63x70xi1>) -> tensor<68x79x131x70xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3, 3, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.tile %0, %t_1 : (tensor<68x79x131x70xi1>, !tosa.shape<4>) -> tensor<204x237x393x140xi1>
    %2 = tosa.minimum %arg2, %arg3 : (tensor<9x29x62x10xf32>, tensor<1x1x1x1xf32>) -> tensor<9x29x62x10xf32>
    %3 = tosa.tanh %2 : (tensor<9x29x62x10xf32>) -> tensor<9x29x62x10xf32>
    return %1, %3 : tensor<204x237x393x140xi1>, tensor<9x29x62x10xf32>
  }
}
