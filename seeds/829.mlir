module {
  func.func @main(%arg0: tensor<93x9x12x52x11x15xf32>) -> tensor<93x9x12x52x11x15xi1> {
    %0 = tosa.log %arg0 : (tensor<93x9x12x52x11x15xf32>) -> tensor<93x9x12x52x11x15xf32>
    %1 = tosa.abs %0 : (tensor<93x9x12x52x11x15xf32>) -> tensor<93x9x12x52x11x15xf32>
    %2 = tosa.ceil %1 : (tensor<93x9x12x52x11x15xf32>) -> tensor<93x9x12x52x11x15xf32>
    %3 = tosa.equal %2, %2 : (tensor<93x9x12x52x11x15xf32>, tensor<93x9x12x52x11x15xf32>) -> tensor<93x9x12x52x11x15xi1>
    return %3 : tensor<93x9x12x52x11x15xi1>
  }
}
