module {
  func.func @main(%arg0: tensor<57x95x58x95x23xi64>) -> tensor<686242950xi64> {
    %r_0 = tosa.const_shape {values = dense<[ 686242950 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<57x95x58x95x23xi64>, !tosa.shape<1>) -> tensor<686242950xi64>
    return %0 : tensor<686242950xi64>
  }
}
