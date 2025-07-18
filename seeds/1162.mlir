module {
  func.func @main(%arg0: tensor<84x84x82x34x94xi64>) -> tensor<1x6x9x10x6xi64> {
    %s_0_start = tosa.const_shape {values = dense<[ 77, 32, 34, 3, 84 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_0_size = tosa.const_shape {values = dense<[ 1, 6, 9, 10, 6 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<84x84x82x34x94xi64>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<1x6x9x10x6xi64>
    return %0 : tensor<1x6x9x10x6xi64>
  }
}
