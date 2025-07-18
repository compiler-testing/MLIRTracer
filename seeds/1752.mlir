module {
  func.func @main(%arg0: tensor<75x66x78x4x22x42xf32>) -> tensor<1427025600xf32> {
    %r_0 = tosa.const_shape {values = dense<[ 1427025600 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<75x66x78x4x22x42xf32>, !tosa.shape<1>) -> tensor<1427025600xf32>
    return %0 : tensor<1427025600xf32>
  }
}
