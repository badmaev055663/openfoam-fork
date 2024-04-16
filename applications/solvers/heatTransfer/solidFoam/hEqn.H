{
    fvScalarMatrix hEqn
    (
        fvm::ddt(betav*rho, h)
      - (
           thermo.isotropic()
         ? fvm::laplacian(betav*thermo.alpha(), h, "laplacian(alpha,h)")
         : fvm::laplacian(betav*taniAlpha(), h, "laplacian(alpha,h)")
        )
      ==
        fvOptions(rho, h)
    );

    if (meshFluxCorr)
    {
        surfaceScalarField phihMesh
        (
            fvc::interpolate(betav*rho*h)*mesh.phi()
        );

       hEqn -= fvc::div(phihMesh);
    }

    hEqn.relax();

    fvOptions.constrain(hEqn);
    double t1 = omp_get_wtime();
    if (useGPU)
        hEqn.solveGPU(opencl);
    else
        hEqn.solve();   //mesh.solver(h.select(finalIter)));
    double t2 = omp_get_wtime();
    printf("solver loop time: %lf ms\n", (t2 - t1) * 1000);

    fvOptions.correct(h);

    thermo.correct();

    Info<< "Min/max T:" << min(thermo.T()).value() << ' '
        << max(thermo.T()).value() << endl;

    radiation->correct();
}
