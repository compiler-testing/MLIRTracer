module {
  func.func @main(%arg0: tensor<89x56x19x32x47x97xi1>, %arg1: tensor<20x40x23xi16>) -> (tensor<89x56x19x32x47x97xi1>, tensor<20x40x1xi16>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<89x56x19x32x47x97xi1>) -> tensor<89x56x19x32x47x97xi1>
    %1 = tosa.reduce_min %arg1 {axis = 2 : i32} : (tensor<20x40x23xi16>) -> tensor<20x40x1xi16>
    %2 = tosa.clz %0 : (tensor<89x56x19x32x47x97xi1>) -> tensor<89x56x19x32x47x97xi1>
    %3 = tosa.clz %1 : (tensor<20x40x1xi16>) -> tensor<20x40x1xi16>
    %4 = tosa.clz %3 : (tensor<20x40x1xi16>) -> tensor<20x40x1xi16>
    return %2, %4 : tensor<89x56x19x32x47x97xi1>, tensor<20x40x1xi16>
  }
}
