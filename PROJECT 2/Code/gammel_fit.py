    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        optimizer=None,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        log_every: int = 1,
        callback=None,
        ):
        """
        Tren nettverket ved å koble sammen compute_gradient -> optimizer.step -> update_weights.
        Importerer ikke optimizere; du sender inn et optimizer-objekt utenfra.

        Args:
            X, Y: Treningsdata (B x D_in), (B x D_out).
            epochs: Antall epoker.
            batch_size: Minibatch-størrelse.
            optimizer: Objekt med signatur step(layers, grads) -> updates.
            X_val, Y_val: (valgfritt) valideringsdatasett for logging.
            shuffle: Om batcher skal shuffles per epoch.
            seed: (valgfritt) seed til shuffling.
            log_every: Logg hver n-te epoch.
            callback: (valgfritt) funksjon f(self, epoch, train_loss, val_loss) kalt etter hver epoch.

        Returns:
            history: dict med lister 'train_loss' og (hvis gitt val) 'val_loss'.
        """
        if optimizer is None:
            raise ValueError("Du må sende inn et optimizer-objekt (f.eks. Adam(lr=1e-3)).")

        rng = np.random.default_rng(seed)
        n = X.shape[0]
        history = {"train_loss": []}
        if X_val is not None and Y_val is not None:
            history["val_loss"] = []

        def iter_minibatches(Xa, Ya, bs, do_shuffle: bool):
            idx = np.arange(Xa.shape[0])
            if do_shuffle:
                rng.shuffle(idx)
            for start in range(0, Xa.shape[0], bs):
                sl = idx[start : start + bs]
                yield Xa[sl], Ya[sl]

        for epoch in range(1, epochs + 1):
            # --- trenings-epoch ---
            for xb, yb in iter_minibatches(X, Y, batch_size, shuffle):
                grads = self.compute_gradient(xb, yb)       # [(dW, db), ...]
                updates = optimizer.step(self.layers, grads) # [(ΔW, Δb), ...]
                self.update_weights(updates)                 # legg til oppdateringene

            # --- logging ---
            train_loss = self.cost(X, Y)
            history["train_loss"].append(train_loss)

            val_loss = None
            if X_val is not None and Y_val is not None:
                val_loss = self.cost(X_val, Y_val)
                history["val_loss"].append(val_loss)

            if (log_every is not None) and (epoch % log_every == 0):
                if val_loss is None:
                    print(f"Epoch {epoch:3d} | train: {train_loss:.6f}")
                else:
                    print(f"Epoch {epoch:3d} | train: {train_loss:.6f} | val: {val_loss:.6f}")

            if callback is not None:
                # callback(self, epoch, train_loss, val_loss)
                callback(self, epoch, train_loss, val_loss)

        return history