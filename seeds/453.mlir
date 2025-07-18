module {
  func.func @main(%arg0: tensor<50x30x94x86xf32>) -> tensor<100x90x94x86xf32> {
    %0 = tosa.identity %arg0 : (tensor<50x30x94x86xf32>) -> tensor<50x30x94x86xf32>
    %t_1 = tosa.const_shape {values = dense<[ 2, 3, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.tile %0, %t_1 : (tensor<50x30x94x86xf32>, !tosa.shape<4>) -> tensor<100x90x94x86xf32>
    %2 = tosa.exp %1 : (tensor<100x90x94x86xf32>) -> tensor<100x90x94x86xf32>
    return %2 : tensor<100x90x94x86xf32>
  }
}
