/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2022 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "PBiCG.H"
#include "PrecisionAdaptor.H"
// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(PBiCG, 0);

    lduMatrix::solver::addasymMatrixConstructorToTable<PBiCG>
        addPBiCGAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::PBiCG::PBiCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::PBiCG::solve
(
    scalarField& psi_s,
    const scalarField& source,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    solveScalarField& psi = tpsi.ref();

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    solveScalar* __restrict__ psiPtr = psi.begin();

    solveScalarField pA(nCells);
    solveScalar* __restrict__ pAPtr = pA.begin();

    solveScalarField wA(nCells);
    solveScalar* __restrict__ wAPtr = wA.begin();

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    ConstPrecisionAdaptor<solveScalar, scalar> tsource(source);
    solveScalarField rA(tsource() - wA);
    solveScalar* __restrict__ rAPtr = rA.begin();

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        true
    );

    // --- Calculate normalisation factor
    const solveScalar normFactor = this->normFactor(psi, tsource(), wA, pA);

    if ((log_ >= 2) || (lduMatrix::debug >= 2))
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
    {
        solveScalarField pT(nCells, 0);
        solveScalar* __restrict__ pTPtr = pT.begin();

        solveScalarField wT(nCells);
        solveScalar* __restrict__ wTPtr = wT.begin();

        // --- Calculate T.psi
        matrix_.Tmul(wT, psi, interfaceIntCoeffs_, interfaces_, cmpt);

        // --- Calculate initial transpose residual field
        solveScalarField rT(tsource() - wT);
        solveScalar* __restrict__ rTPtr = rT.begin();

        // --- Initial value not used
        solveScalar wArT = 0;

        // --- Select and construct the preconditioner
        if (!preconPtr_)
        {
            preconPtr_ = lduMatrix::preconditioner::New
            (
                *this,
                controlDict_
            );
        }

        // --- Solver iteration
        do
        {
            // --- Store previous wArT
            const solveScalar wArTold = wArT;

            // --- Precondition residuals
            preconPtr_->precondition(wA, rA, cmpt);
            preconPtr_->preconditionT(wT, rT, cmpt);

            // --- Update search directions:
            wArT = gSumProd(wA, rT, matrix().mesh().comm());

            if (solverPerf.nIterations() == 0)
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                    pTPtr[cell] = wTPtr[cell];
                }
            }
            else
            {
                const solveScalar beta = wArT/wArTold;

                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta*pAPtr[cell];
                    pTPtr[cell] = wTPtr[cell] + beta*pTPtr[cell];
                }
            }


            // --- Update preconditioned residuals
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            matrix_.Tmul(wT, pT, interfaceIntCoeffs_, interfaces_, cmpt);

            const solveScalar wApT = gSumProd(wA, pT, matrix().mesh().comm());

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApT)/normFactor))
            {
                break;
            }


            // --- Update solution and residual:

            const solveScalar alpha = wArT/wApT;

            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pAPtr[cell];
                rAPtr[cell] -= alpha*wAPtr[cell];
                rTPtr[cell] -= alpha*wTPtr[cell];
            }

            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_, log_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    // Recommend PBiCGStab if PBiCG fails to converge
    const label upperMaxIters = max(maxIter_, lduMatrix::defaultMaxIter);

    if (solverPerf.nIterations() > upperMaxIters)
    {
        FatalErrorInFunction
            << "PBiCG has failed to converge within the maximum iterations: "
            << upperMaxIters << nl
            << "    Please try the more robust PBiCGStab solver."
            << exit(FatalError);
    }

    if (preconPtr_)
    {
        preconPtr_->setFinished(solverPerf);
    }

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        false
    );

    return solverPerf;
}

Foam::solverPerformance Foam::PBiCG::solveGPU
(
    scalarField& psi_s,
    const scalarField& source,
    OpenCL& opencl,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    solveScalarField& psi = tpsi.ref();

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    cl::Kernel copyKernel(opencl.program, "copy");
    cl::Kernel sumProdKernel(opencl.program, "sumProd");
    cl::Kernel multAddKernel(opencl.program, "multAdd");
    cl::Kernel addMultKernel(opencl.program, "addMult");
    cl::Kernel sumMagKernel(opencl.program, "sumMag");
    cl::Kernel invKernel(opencl.program, "inverse");
    cl::Kernel multKernel(opencl.program, "mult");

    solveScalar* __restrict__ psiPtr = psi.begin();

    solveScalarField pA(nCells);
    solveScalar* __restrict__ pAPtr = pA.begin();

    solveScalarField wA(nCells);
    solveScalar* __restrict__ wAPtr = wA.begin();

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    ConstPrecisionAdaptor<solveScalar, scalar> tsource(source);
    solveScalarField rA(tsource() - wA);
    solveScalar* __restrict__ rAPtr = rA.begin();

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        true
    );

    // --- Calculate normalisation factor
    const solveScalar normFactor = this->normFactor(psi, tsource(), wA, pA);

    if ((log_ >= 2) || (lduMatrix::debug >= 2))
    {
        Info<< "   Normalisation factor = " << normFactor << endl;
    }
    cl::Buffer rA_buf(opencl.queue, rAPtr, rAPtr + nCells, false);

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        sumMagGPU(opencl, sumMagKernel, rA_buf, nCells) /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
    {
        solveScalarField pT(nCells, 0);
        solveScalar* __restrict__ pTPtr = pT.begin();

        solveScalarField wT(nCells);
        solveScalar* __restrict__ wTPtr = wT.begin();

        // --- Calculate T.psi
        matrix_.Tmul(wT, psi, interfaceIntCoeffs_, interfaces_, cmpt);

        // --- Calculate initial transpose residual field
        solveScalarField rT(tsource() - wT);
        solveScalar* __restrict__ rTPtr = rT.begin();

        // --- Initial value not used
        solveScalar wArT = 0;
        cl::Buffer wA_buf(opencl.queue, wAPtr, wAPtr + nCells, false);
        cl::Buffer wT_buf(opencl.queue, wTPtr, wTPtr + nCells, false);

        cl::Buffer pA_buf(opencl.queue, pAPtr, pAPtr + nCells, false);
        cl::Buffer pT_buf(opencl.queue, pTPtr, pTPtr + nCells, false);
        cl::Buffer psi_buf(opencl.queue, psiPtr, psiPtr + nCells, false);

        cl::Buffer rT_buf(opencl.queue, rTPtr, rTPtr + nCells, false);

        scalar* const __restrict__ diagPtr = const_cast<scalar*>(matrix_.diag().begin());

        cl::Buffer diag_buf(opencl.queue, diagPtr, diagPtr + nCells, true);

        label* const __restrict__ uPtr = const_cast<label*>(matrix_.lduAddr().upperAddr().begin());
        label* const __restrict__ lPtr = const_cast<label*>(matrix_.lduAddr().lowerAddr().begin());

        scalar* const __restrict__ upperPtr = const_cast<scalar*>(matrix_.upper().begin());
        scalar* const __restrict__ lowerPtr = const_cast<scalar*>(matrix_.lower().begin());

        const label nFaces = matrix_.upper().size();
            
        cl::Buffer lower_buf(opencl.queue, lowerPtr, lowerPtr + nFaces, true);
        cl::Buffer upper_buf(opencl.queue, upperPtr, upperPtr + nFaces, true);

        cl::Buffer l_buf(opencl.queue, lPtr, lPtr + nFaces, true);
        cl::Buffer u_buf(opencl.queue, uPtr, uPtr + nFaces, true);

        cl::Buffer rD_buf(opencl.queue, diagPtr, diagPtr + nCells, false);
        bool usePrecond = true;
        if (usePrecond) { // init diagonal precondition
            invKernel.setArg(0, rD_buf);
            invKernel.setArg(1, nCells);
            opencl.queue.enqueueNDRangeKernel(invKernel, cl::NullRange,
                        cl::NDRange(nCells - nCells % locSz), cl::NDRange(locSz));
        }
        opencl.queue.finish();
        // --- Solver iteration
        do
        {
            // --- Store previous wArT
            const solveScalar wArTold = wArT;
            if (usePrecond) { // diagonal precondition
                diagPrecondGPU(opencl, multKernel, wA_buf, rA_buf, rD_buf, nCells);
                diagPrecondGPU(opencl, multKernel, wT_buf, rT_buf, rD_buf, nCells);
            } else { // none precondition
                copyGPU(opencl, copyKernel, wA_buf, rA_buf, nCells);
                copyGPU(opencl, copyKernel, wT_buf, rT_buf, nCells);
            }
            opencl.queue.finish();

            // --- Update search directions:
            wArT = sumProdGPU(opencl, sumProdKernel, wA_buf, rT_buf, nCells);

            if (solverPerf.nIterations() == 0)
            {
                copyGPU(opencl, copyKernel, pA_buf, wA_buf, nCells);
                copyGPU(opencl, copyKernel, pT_buf, wT_buf, nCells);
                opencl.queue.finish();
            }
            else
            {
                const solveScalar beta = wArT/wArTold;
                multAddKernel.setArg(0, pA_buf);
                multAddKernel.setArg(1, wA_buf);
                multAddKernel.setArg(2, beta);
                multAddKernel.setArg(3, nCells);
                opencl.queue.enqueueNDRangeKernel(multAddKernel, cl::NullRange,
                                            cl::NDRange(nCells - nCells % locSz), cl::NDRange(locSz));

                multAddKernel.setArg(0, pT_buf);
                multAddKernel.setArg(1, wT_buf);
                multAddKernel.setArg(2, beta);
                multAddKernel.setArg(3, nCells);
                opencl.queue.enqueueNDRangeKernel(multAddKernel, cl::NullRange,
                                            cl::NDRange(nCells - nCells % locSz), cl::NDRange(locSz));
                opencl.queue.finish();
            }
            // --- Update preconditioned residuals
            matrix_.AmulGPU(opencl, wA_buf, pA_buf, diag_buf, lower_buf, upper_buf, l_buf, u_buf);
            matrix_.TmulGPU(opencl, wT_buf, pT_buf, diag_buf, lower_buf, upper_buf, l_buf, u_buf);

            const solveScalar wApT = sumProdGPU(opencl, sumProdKernel, wA_buf, pT_buf, nCells);

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApT)/normFactor))
            {
                break;
            }

            // --- Update solution and residual:

            const solveScalar alpha = wArT/wApT;
            addMultGPU(opencl, addMultKernel, pA_buf, psi_buf, alpha, nCells);
            addMultGPU(opencl, addMultKernel, wA_buf, rA_buf, -alpha, nCells);
            addMultGPU(opencl, addMultKernel, wT_buf, rT_buf, -alpha, nCells);
            opencl.queue.finish();

            solverPerf.finalResidual() =
                sumMagGPU(opencl, sumMagKernel, rA_buf, nCells) / normFactor;
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_, log_)
            )
         || solverPerf.nIterations() < minIter_
        );
        opencl.queue.enqueueReadBuffer(psi_buf, false, 0, sizeof(double) * nCells, psiPtr);
        opencl.queue.enqueueReadBuffer(rA_buf, false, 0, sizeof(double) * nCells, rAPtr);
        opencl.queue.enqueueReadBuffer(wA_buf, false, 0, nCells * sizeof(double), wAPtr);
        opencl.queue.finish();
    }

    // Recommend PBiCGStab if PBiCG fails to converge
    const label upperMaxIters = max(maxIter_, lduMatrix::defaultMaxIter);

    if (solverPerf.nIterations() > upperMaxIters)
    {
        FatalErrorInFunction
            << "PBiCG has failed to converge within the maximum iterations: "
            << upperMaxIters << nl
            << "    Please try the more robust PBiCGStab solver."
            << exit(FatalError);
    }

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        false
    );

    return solverPerf;
}


// ************************************************************************* //
