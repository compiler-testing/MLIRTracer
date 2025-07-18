module {
  func.func @main(%arg0: tensor<91x94x88x72xi16>, %arg1: tensor<91x94x1x72xi16>, %arg2: tensor<12x6xi1>, %arg3: tensor<98x34x48xi32>, %arg4: tensor<98x34x1xi32>) -> (tensor<91x94x88x72xi16>, tensor<12x6xi1>, tensor<98x34x48xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<91x94x88x72xi16>, tensor<91x94x1x72xi16>) -> tensor<91x94x88x72xi16>
    %1 = tosa.logical_not %arg2 : (tensor<12x6xi1>) -> tensor<12x6xi1>
    %2 = tosa.logical_not %1 : (tensor<12x6xi1>) -> tensor<12x6xi1>
    %3 = tosa.greater_equal %arg3, %arg4 : (tensor<98x34x48xi32>, tensor<98x34x1xi32>) -> tensor<98x34x48xi1>
    return %0, %2, %3 : tensor<91x94x88x72xi16>, tensor<12x6xi1>, tensor<98x34x48xi1>
  }
}
