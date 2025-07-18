module {
  func.func @main(%arg0: tensor<46x95x63x27x94x83xf32>) -> tensor<46x95x63x27x94x83xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<46x95x63x27x94x83xf32>) -> tensor<46x95x63x27x94x83xf32>
    return %0 : tensor<46x95x63x27x94x83xf32>
  }
}
